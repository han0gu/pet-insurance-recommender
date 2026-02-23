from langchain_core.documents import Document

chunk = Document(
    page_content=('- 19. 첩모난생(속눈썹 질환) 및 눈물샘 치료(누루관시술 등) 등의 안검 외·내반 및\n'
 '- 비루관 관련 질환으로 인한 비용\n'
 '- \uf000 제2항에서 정한 조치에 다른 진료를 병행하여 실시한 경우에는 제2항에서 정한 조\n'
 '- 치(마취 비용을 포함합니다.)에 대한 보험금은 지급하지 않습니다.\n'
 '제5조(보험금의 청구)\uf000 보험수익자는다음의 서류를 제출하고 보험금을 청구하여야 합니다.1. 청구서(회사 양식)- 2. 국가동물 '
 '등록한 경우에는 동물등록증 또는 등록번호\n'
 '- 3. 국가동물 미등록한 경우에는 가입동물의 사진 2매(얼굴전면, 측면전신사진)를'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
