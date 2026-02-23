from langchain_core.documents import Document

chunk = Document(
    page_content=('- 상책임\n'
 '- 15. 가입동물의 소음, 냄새, 털날림으로 인하여 발생한 배상책임\n'
 '- 16. 가입동물이 질병을 전염시켜 발생한 배상책임\n'
 '- 17. 피보험자의 피용인이 피보험자의 업무에 종사 중에 입은 신체의 장해(상해, 질병 및 그로 인한\n'
 '- 사망을 말합니다)에 기인하는 배상책임\n'
 '- 18. 동물보호법 시행규칙 제1조의 2에 따른 맹견의 경우 동법 시행규칙 제12조 제2항에 따라 목줄\n'
 '- 과 입마개를 하지 않아 발생한 손해에 대한 배상책임\n'
 '【핵연료물질】 사용된 연료를 포함합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
