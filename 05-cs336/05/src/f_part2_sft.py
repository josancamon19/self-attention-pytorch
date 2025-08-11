import random
from src.b_sft import init_wandb, get_model_and_tokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import json

# {
#     "prompt": "How does the author propose to fix the problem of science alienation in our educational system? What changes does she suggest should be made to science education? Answer according to: Science education should be split into two tracks.\nSplit K-12 science education into two tracks, for majors and nonmajors.\nThose who want to specialize in science could take math and complex chemistry. Nonmajors would focus on science of the everyday\u2014things like kitchen chemistry and CSI-style crime investigations.\nSome years ago, when I was working as a newspaper science writer in California, I fell into a rather idle conversation with a physicist on the subject of science education. Idle for him, at least, because what he said\u2014the way he defined the American system of teaching K-12 students about science\u2014has stayed with me since.\nThis conversation has returned to me many times over the years. On the day my older son, who spent his childhood joyfully chasing insects and reading natural history books, came home from high school to announce that \u201cbiology is the most boring subject in the world.\u201d The time the student adviser at the University of Wisconsin journalism school, where I teach, told me that undergrads don\u2019t want to take science-writing classes because they already felt \u201cbeaten up by science.\u201d Even on days when I read science literacy statistics and am startled by the percentage of people who think that antibiotics kill viruses as well as bacteria or that the sun revolves around the Earth.\nIt\u2019s easy for me to join with the chorus of concern over whether we are keeping up as a science-literate country. But even so, science literacy is the wrong term for the worried point I\u2019m making here. I\u2019m concerned about science alienation. The way we teach K-12 science, with its emphasis on the intricacies of formula and mathematical structure necessary for the next generation of scientists and engineers, shuts too many people out. I don\u2019t mean just shut out of the priesthood here; there\u2019s no reason for everyone to become a scientist. But shut out of the comfort zone, of the ability to think of science as something useful in daily life.\nAnd it\u2019s the science-alienated who matter to me. Partly because as a science writer I find them the most interesting audience\u2014I love the idea of seducing someone convinced that science is boring or meaningless into reading a research-centered story. Partly because I worry that what The New Yorker\u2019s Michael Specter describes as a culture of science denialism is having a profound\u2014possibly dangerous\u2014effect on public policies. Think, for instance, of recent moves in Virginia, Texas, and North Carolina to deny that sea level rise is related to global climate change. And partly I just think that the filtered-out are cheated. They endure the mandatory classes, but they never really learn the main lesson\u2014the way that science helps us understand ourselves and the world around us.\nThe great psychologist and philosopher William James wrote more than 100 years ago that the science establishment would face consequences if it dismissed the importance of the lay public. In the year 1909, he said, it was easy to find people who no longer paid attention to the teachings of researchers. \u201cThey are indifferent to science,\u201d he explained, \u201cbecause science is so callously indifferent to their experiences.\u201d Today, this is very relevant to K-12 science education: Our system breeds that indifference.\nHow can we fix it? By requiring even more science education\u2014but changing it dramatically. All high school students should have to take four years of science, instead of the minimal two usually required for graduation. Along with that, we should also offer separate \u201cscience tracks\u201d\u2014something like the old tracks for vocational or college-bound students. Those who want to specialize in science could take math, physics, and complex chemistry (and the kind of genetically detailed biology that my son disliked). Nonmajors, by contrast, would focus on what I think of as science of the everyday\u2014things like kitchen chemistry, CSI-style crime investigation, the biology of health, and a class in the physics of sports that would include playing the games.\nMy tracking system could easily build on existing science electives. My son\u2019s school, for instance, teaches a forensic science unit which uses the classroom as an ever changing \u201ccrime scene,\u201d exploring everything from DNA analysis to blood typing. It\u2019s a way of putting science in context. More students would take such classes; more would be exposed to these ideas as they did the coursework needed to complete each track. Ideally, the tracks could be coordinated with the science classes offered for majors and nonmajors at the college level. Of course, occasionally a student might want to switch tracks\u2014and we should allow for that, too. It could go both ways: Would-be scientists might realize that the field isn\u2019t really for them and switch to the nonmajor track, while some students might be inspired by the CSI class to move into the more difficult sciences.\nWe need to stop trying to teach every child as either a future scientist or a future failed scientist. We don\u2019t want or need every student to be a scientist, but we do want each one to be a success. And success includes a solid understanding and appreciation of science, one that will remain useful to both themselves and society throughout their lives.\nAlso in Slate\u2019s special issue on science education: Fred Kaplan explains why another \u201cSputnik moment\u201d would be impossible; Philip Plait explains why he became the \u201cBad Astronomer\u201d; Paul Plotz describes how almost blowing up his parents\u2019 basement made him a scientist; Tom Kalil says that the Obama administration is using the Make movement to encourage science education; and Dana Goldstein explains why you should make your daughter play video games. Also, share your ideas for fixing science education in the Hive. This article arises from Future Tense, a joint partnership of Slate, the New America Foundation, and Arizona State University.",
#     "response": "The author proposes to fix the problem of science alienation in our educational system by splitting K-12 science education into two tracks, for majors and nonmajors. Those who want to specialize in science could take math and complex chemistry, while nonmajors would focus on science of the everyday \u2013 things like kitchen chemistry and CSI-style crime investigations. All high school students should take four years of science, and the tracking system could build on existing science electives. The goal is to teach every student a solid understanding and appreciation of science, one that will remain useful to both themselves and society throughout their lives.",
# }

# {
#     "prompt": "Rice tolerance to suboptimal low temperatures relies on the maintenance of the photosynthetic capacity.\nGazquez, A., Vilas, J. M., Colman Lerner, J. E., Maiale, S. J., Calzadilla, P. I., Menendez, A. B. And Rodriguez, A. A.\nLaboratorio de Fisiologia de Estres Abiotico en Plantas, Unidad de Biotecnologia 1, IIB-INTECH, CONICET, UNSAM, Chascomus, Argentina.\nCentro de Investigaciones y Desarrollo en Ciencias Aplicadas, FCEx, UNLP, Argentina.\nDepartamento de Biodiversidad y Biologia Experimental, FCEyN - UBA, INMIBO-CONICET, Buenos Aires, Argentina.\nLaboratorio de Fisiologia de Estres Abiotico en Plantas, Unidad de Biotecnologia 1, IIB-INTECH, CONICET, UNSAM, Chascomus, Argentina. Electronic address: andresrodriguez@conicet.gov.ar.\nThe purpose of this research was to identify differences between two contrasting rice cultivars in their response to suboptimal low temperatures stress. A transcriptomic analysis of the seedlings was performed and results were complemented with biochemical and physiological analyses. The microarray analysis showed downregulation of many genes related with PSII and particularly with the oxygen evolving complex in the sensitive cultivar IR50. Complementary studies indicated that the PSII performance, the degree of oxygen evolving complex coupling with the PSII core and net photosynthetic rate diminished in this cultivar in response to the stress. However, the tolerant cultivar Koshihikari was able to maintain its energy equilibrium by sustaining the photosynthetic capacity. The increase of oleic acid in Koshihikari could be related with membrane remodelling of the chloroplasts and hence contribute to tolerance. Overall, these results work as a ground for future analyses that look forward to characterize possible mechanisms to tolerate this stress. How did the tolerant rice cultivar Koshihikari maintain its photosynthetic capacity under suboptimal low temperatures stress, and what possible mechanisms could contribute to its tolerance?",
#     "response": "The tolerant rice cultivar Koshihikari maintained its photosynthetic capacity under suboptimal low temperatures stress by sustaining its energy equilibrium. This was achieved through maintaining the activity of the photosystem II (PSII) and the coupling of the oxygen evolving complex with the PSII core, resulting in a higher net photosynthetic rate. Additionally, Koshihikari exhibited an increase in oleic acid, which could potentially contribute to membrane remodeling of chloroplasts and contribute to tolerance. Possible mechanisms that could also contribute to Koshihikari's tolerance include increased antioxidant defense mechanisms and changes in the expression of stress-responsive genes. Further research is necessary to fully understand how Koshihikari is able to tolerate suboptimal low temperatures stress and to identify potential targets for breeding programs aimed at improving rice yields under adverse environmental conditions.",
# }


class UltraChatDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        dataset_path: str,
        seq_length: int = 512,
        shuffle=False,
        subset: str = "train",
    ):
        super().__init__()

        self.seq_length = seq_length
        self.shuffle = shuffle
        self.tokenizer = tokenizer

        with open("src/prompts/alpaca_sft.prompt") as f:
            self.prompt_template = f.read().strip()

        if not dataset_path:
            dataset_path = (
                f"data/safety_augmented_ultrachat_200k_single_turn/{subset}.jsonl"
            )

        with open(dataset_path, "r") as f:
            self.dataset = [json.loads(line.strip()) for line in f]

        if shuffle:
            random.shuffle(self.dataset)

        contents = ""
        for i, item in enumerate(self.dataset):
            print("loading_item")
            prompt = self.prompt_template.format(
                instruction=item["prompt"], response=item["response"]
            )
            contents += prompt + tokenizer.eos_token

        self.tokenized = self.tokenizer.encode(
            contents, padding=False, return_tensors="pt", add_special_tokens=True
        )[0]

        self.count = (len(self.tokenized) - 1) // self.seq_length

    def __len__(self):
        return self.count

    def __getitem__(self, i):
        start_idx = i * self.seq_length
        end_idx = start_idx + self.seq_length
        chunk = self.tokenized[start_idx : end_idx + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def run_sft():
    init_wandb()
    model, tokenizer = get_model_and_tokenizer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
