# Third-party components

The repository's MIT License covers material that the copyright holder owns or
has permission to publish. Packages installed from the requirement files are
external dependencies, are not vendored in this repository, and remain subject
to their respective licenses. Review the license metadata of the exact resolved
environment before redistributing a bundled application or container.

The Docker definitions reference official Python and NVIDIA CUDA base images.
Those images and the operating-system packages they contain remain under their
respective terms. Before publishing a built image, review the resolved image,
package licenses, and applicable NVIDIA container license terms; generate an
up-to-date software bill of materials for the artifact being distributed.

`true_dna/dna_model/bpe_vocab/tokenizer.json` is a generated tokenizer asset in
the Hugging Face Tokenizers serialization format. It contains vocabulary,
merge, and special-token configuration; it does not contain model weights or
complete source FASTA records. Because its vocabulary and merge rules are
derived from a training corpus, releases should retain corpus provenance and
confirm the right to distribute the derived asset. Future replacements should
document tokenizer settings and compatibility impact as well.

The repository does not bundle downloaded genomic datasets or pretrained model
weights. Data retrieved from NCBI or another provider remains subject to that
provider's terms and any restrictions attached to the underlying records.
