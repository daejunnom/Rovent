"""Pure N05 metadata/loss fixtures; no model, optimizer or actual source cap."""
import copy
import math
import unittest

from rz_pals_model import rovent_utility_training as training


def frozen_config():
    return {"board_width":2,"line_width":3,"context_width":4,"local_width":5,
            "hidden_width":6,"causal_window":3,"max_lines":64,"max_plies":256}


def membership(*, owner="utility-adamw-owner", hidden=2):
    return {
        "hidden_width":hidden,"input_width":17,
        "wire_to_parameter":{name:"utility."+name for name in training.PARAMETER_FIELDS},
        "parameters":{name:{"checkpoint_path":"parameters."+name,"shape":shape,
                            "dtype":"float32","planned_requires_grad":True}
                      for name,shape in zip(training.PARAMETER_FIELDS,([hidden,17],[hidden],[7,hidden],[7]),strict=True)},
        "optimizer_owner":owner,"frozen_components":["nnuec_base","p","c"],
    }


def registration(spec):
    return {
        "collection_source_sha256":"10"*32,"trainer_source_sha256":"11"*32,
        "resource_profile_sha256":"40"*32,"cost_epoch_sha256":"50"*32,
        "cost_scope":"exclusive_submitted_global_target_route_and_descendants_with_required_drain",
        "objective":"exclusive_wall_nanoseconds",
        "frozen_base":{"schema":training.BASE_SCHEMA,"semantics":training.BASE_SEMANTICS,
                       "weights_sha256":"20"*32,"config":frozen_config()},
        "utility":{"schema":training.UTILITY_SCHEMA,"semantics":training.UTILITY_SEMANTICS,
                   "initial_weights_sha256":"30"*32,"base_weights_sha256":"20"*32,
                   "parameter_spec_sha256":training.parameter_spec_sha256(spec,frozen_config(),spec["optimizer_owner"])},
    }


class UtilityRecipePreparationTests(unittest.TestCase):
    def test_new_recipe_records_explicit_loss_and_fixed_optimizer_as_decimal_strings(self):
        recipe=training.make_recipe("utility-adamw-owner")
        self.assertEqual(recipe["schema"],"rz-rovent-utility-training-recipe/1")
        self.assertEqual(recipe["loss"],{"kind":"pairwise_logistic","route_actions":[4,5,6],
                                        "normalization":"mean_unique_known_pairs","weight":"1.0"})
        self.assertEqual(recipe["optimizer"],{"kind":"adamw","learning_rate":"0.0001","betas":["0.9","0.999"],
            "epsilon":"0.00000001","weight_decay":"0.01","gradient_clip_norm":"1.0","schedule":"constant"})
        self.assertEqual(training.validate_recipe(recipe),recipe)
        self.assertEqual(len(training.recipe_sha256(recipe)),64)

    def test_recipe_rejects_legacy_domains_changed_defaults_and_numeric_aliases(self):
        original=training.make_recipe("utility-adamw-owner")
        for mutate in (lambda r:r.update(schema="rz-pals-training-recipe/1"),
                       lambda r:r["optimizer"].update(learning_rate=0.0001),
                       lambda r:r["loss"].update(weight="0.5"),
                       lambda r:r["loss"].update(route_actions=[4.0,5,6]),
                       lambda r:r["limits"].update(max_contexts=8192.0)):
            changed=copy.deepcopy(original);mutate(changed)
            with self.assertRaises(ValueError):training.validate_recipe(changed)

    def test_only_four_private_fp32_arrays_join_the_actual_frozen_base_config(self):
        spec=membership()
        checked=training.validate_parameter_membership(spec,frozen_config(),"utility-adamw-owner")
        self.assertEqual(checked["input_width"],2*2+2*4+5)
        self.assertEqual(checked["parameters"]["output_weights"]["shape"],[7,2])
        self.assertEqual(checked["parameters"]["hidden_weights"]["checkpoint_path"],"parameters.hidden_weights")
        self.assertEqual(checked["wire_to_parameter"]["hidden_weights"],"utility.hidden_weights")
        self.assertEqual(spec,membership())

    def test_membership_rejects_extra_base_arrays_aliases_dtype_shape_owner_and_unfreeze(self):
        for mutate in (lambda s:s["parameters"].update(board_weights={}),
                       lambda s:s["wire_to_parameter"].update(hidden_bias=s["wire_to_parameter"]["hidden_weights"]),
                       lambda s:s["wire_to_parameter"].update(hidden_weights="base.hidden_weights"),
                       lambda s:s["parameters"]["hidden_weights"].update(dtype="float16"),
                       lambda s:s["parameters"]["hidden_weights"].update(shape=[2,16]),
                       lambda s:s.update(optimizer_owner="pc-shared-owner"),
                       lambda s:s.update(frozen_components=[])):
            changed=membership();mutate(changed)
            with self.assertRaises(ValueError):
                training.validate_parameter_membership(changed,frozen_config(),"utility-adamw-owner")
        wrong=frozen_config();wrong["board_width"]=3
        with self.assertRaises(ValueError):training.validate_parameter_membership(membership(),wrong,"utility-adamw-owner")

    def test_known_pairs_separate_unique_comparisons_occurrences_and_masked_entries(self):
        pairs=[[6,4],None,[6,4],[5,4]]
        result=training.normalize_known_pairs(pairs)
        self.assertEqual(result["pairs"],[[5,4],[6,4]])
        self.assertEqual(result["occurrence_counts"],[1,2])
        self.assertEqual(result["unique_known_pairs"],2)
        self.assertEqual(result["known_occurrences"],3)
        self.assertEqual(result["masked_entries"],1)
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(pairs,[[6,4],None,[6,4],[5,4]])

    def test_contradictory_self_unsupported_and_boolean_comparisons_are_rejected(self):
        for pairs in ([[4,5],[5,4]],[[4,4]],[[3,4]],[[4,True]],[[4,7]],[[4,5,6]]):
            with self.assertRaises(ValueError):training.normalize_known_pairs(pairs)
        with self.assertRaises(ValueError):
            training.normalize_known_pairs([[4,5]]*(training.MAX_PAIR_ENTRIES_PER_CONTEXT+1))

    def test_loss_is_mean_of_unique_known_route_pairs_not_repeated_occurrences(self):
        logits=[[0,0,0,0,1.0,0.0,0.0],[0,0,0,0,0.0,0.0,0.0]]
        summary=training.route_pairwise_loss(logits,[[[4,5],[4,5],None],[[6,5]]])
        expected=math.log1p(math.exp(-1.0))+math.log(2.0)
        self.assertAlmostEqual(summary["loss_sum"],expected)
        self.assertAlmostEqual(summary["normalized_loss"],expected/2)
        self.assertEqual(summary["known_pair_denominator"],2)
        self.assertEqual(summary["known_occurrences"],3)
        self.assertEqual(summary["performed_optimizer_updates"],0)
        self.assertFalse(summary["execution_authorized"])

    def test_all_masked_has_no_mean_or_optimizer_update(self):
        summary=training.route_pairwise_loss([[0.0]*7],[[None,None]])
        self.assertEqual(summary["known_pair_denominator"],0)
        self.assertEqual(summary["loss_sum"],0.0)
        self.assertIsNone(summary["normalized_loss"])
        self.assertEqual(summary["performed_optimizer_updates"],0)
        self.assertIsNone(training.route_pairwise_loss([],[])["normalized_loss"])

    def test_scalar_reference_is_stable_and_rejects_nonfinite_or_misaligned_rows(self):
        result=training.route_pairwise_loss([[0,0,0,0,1000.0,-1000.0,0]],[[[4,5]]])
        self.assertTrue(math.isfinite(result["loss_sum"]))
        for logits,pairs in (([[0.0]*7],[]),([[0.0]*6],[[[4,5]]]),
                             ([[0,0,0,0,float("nan"),0,0]],[[[4,5]]]),
                             ([[0,0,0,0,10**1000,0,0]],[[[4,5]]])):
            with self.assertRaises(ValueError):training.route_pairwise_loss(logits,pairs)

    def test_preparation_seals_source_cost_model_and_spec_but_creates_no_tensor_or_authority(self):
        spec=membership();registered=registration(spec);recipe=training.make_recipe(spec["optimizer_owner"])
        before=copy.deepcopy((recipe,registered,spec))
        result=training.prepare_training(recipe,registered,spec)
        self.assertEqual(result["schema"],"rz-rovent-utility-training-preparation/1")
        self.assertEqual(result["registration"]["collection_source_sha256"],"10"*32)
        self.assertEqual(result["registration"]["trainer_source_sha256"],"11"*32)
        self.assertEqual(result["registration"]["frozen_base"]["weights_sha256"],"20"*32)
        self.assertEqual(result["registration"]["utility"]["initial_weights_sha256"],"30"*32)
        self.assertEqual(result["registration"]["resource_profile_sha256"],"40"*32)
        self.assertEqual(result["registration"]["cost_epoch_sha256"],"50"*32)
        self.assertEqual(result["registration"]["cost_scope"],
                         "exclusive_submitted_global_target_route_and_descendants_with_required_drain")
        self.assertEqual(result["registration"]["objective"],"exclusive_wall_nanoseconds")
        self.assertNotIn("cost_epoch",result["registration"])
        self.assertNotIn("source_sha256",result["registration"])
        self.assertEqual(len({registered["collection_source_sha256"],registered["trainer_source_sha256"],
                              registered["frozen_base"]["weights_sha256"],
                              registered["utility"]["initial_weights_sha256"],registered["resource_profile_sha256"],
                              registered["cost_epoch_sha256"]}),6)
        self.assertEqual(result["prepared_artifacts"],["checkpoint_metadata","export_metadata"])
        for flag in ("tensor_membership_verified","source_authority_verified","optimizer_state_present",
                     "checkpoint_tensor_present","export_tensor_present","execution_authorized"):
            self.assertFalse(result[flag])
        self.assertEqual(result["performed_optimizer_updates"],0)
        self.assertEqual(training.prepare_training(recipe,registered,spec),result)
        self.assertEqual((recipe,registered,spec),before)

    def test_wrong_pins_and_changed_cost_resource_digests_are_rejected_or_change_the_seal(self):
        spec=membership();registered=registration(spec);recipe=training.make_recipe(spec["optimizer_owner"])
        original=training.prepare_training(recipe,registered,spec)
        for field,value in (("collection_source_sha256","12"*32),("trainer_source_sha256","13"*32),
                            ("cost_epoch_sha256","51"*32),("resource_profile_sha256","41"*32)):
            changed=copy.deepcopy(registered);changed[field]=value
            self.assertNotEqual(training.prepare_training(recipe,changed,spec)["registration_sha256"],original["registration_sha256"])
            self.assertNotEqual(training.prepare_training(recipe,changed,spec)["preparation_sha256"],original["preparation_sha256"])
        swapped=copy.deepcopy(registered)
        swapped["cost_epoch_sha256"],swapped["resource_profile_sha256"]=registered["resource_profile_sha256"],registered["cost_epoch_sha256"]
        self.assertNotEqual(training.prepare_training(recipe,swapped,spec)["preparation_sha256"],original["preparation_sha256"])
        for mutate in (lambda r:r.update(cost_epoch=3),
                       lambda r:r.update(cost_epoch_sha256=3),
                       lambda r:r.update(resource_profile_sha256=0),
                       lambda r:r.update(cost_scope="root_wall_interval_only"),
                       lambda r:r.update(objective="cpu_nanoseconds"),
                       lambda r:r["utility"].update(base_weights_sha256="99"*32),
                       lambda r:r["utility"].update(parameter_spec_sha256="99"*32),
                       lambda r:r["frozen_base"].update(semantics="invented-actual-authority"),
                       lambda r:r.update(collection_source_sha256="AB"*32),
                       lambda r:r.update(trainer_source_sha256="AB"*32)):
            changed=copy.deepcopy(registered);mutate(changed)
            with self.assertRaises(ValueError):training.prepare_training(recipe,changed,spec)

    def test_old_integer_epoch_is_not_automatically_converted_or_mixed_with_new_scope(self):
        spec=membership();registered=registration(spec);recipe=training.make_recipe(spec["optimizer_owner"])
        legacy={"source_sha256":registered["collection_source_sha256"],"cost_epoch":3,
                "frozen_base":registered["frozen_base"],"utility":registered["utility"]}
        with self.assertRaises(ValueError):training.prepare_training(recipe,legacy,spec)
        for missing in ("cost_epoch_sha256","resource_profile_sha256","cost_scope","objective"):
            changed=copy.deepcopy(registered);changed.pop(missing)
            with self.assertRaises(ValueError):training.prepare_training(recipe,changed,spec)

    def test_collection_and_trainer_sources_have_no_legacy_alias_or_automatic_fill(self):
        spec=membership();registered=registration(spec);recipe=training.make_recipe(spec["optimizer_owner"])
        for missing in ("collection_source_sha256","trainer_source_sha256"):
            changed=copy.deepcopy(registered);changed.pop(missing)
            with self.assertRaises(ValueError):training.prepare_training(recipe,changed,spec)
            changed["source_sha256"]=registered[missing]
            with self.assertRaises(ValueError):training.prepare_training(recipe,changed,spec)
        changed=copy.deepcopy(registered)
        changed["source_sha256"]=registered["collection_source_sha256"]
        with self.assertRaises(ValueError):training.prepare_training(recipe,changed,spec)
        changed=copy.deepcopy(registered)
        changed["source_sha256"]=changed.pop("collection_source_sha256")
        changed.pop("trainer_source_sha256")
        with self.assertRaises(ValueError):training.prepare_training(recipe,changed,spec)


if __name__ == "__main__":
    unittest.main()
