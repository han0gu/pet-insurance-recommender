from langchain_core.documents import Document

chunk = Document(
    page_content=('- ·보상비율 70%, 자기부담금 3만원, 기본형Ⅱ 가입 기준\n'
 '·MRI/CT# 시행 시 보상한도액 : 100만원 한도- 예시①\n'
 '- ·MRI/CT 시행 당일 피보험자가 부담한 의료비 : 78만원\n'
 '- ·반려동물의료비보험금 : 15만원\n'
 '- ·지급금액 = {(78만원 - 15만원 - 3만원) x 70%, 100만원} 중 적은 금액\n'
 '# = 42만원- 예시②\n'
 '- ·MRI/CT 시행 당일 피보험자가 부담한 의료비 : 218만원\n'
 '- ·반려동물의료비보험금 : 15만원'),
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
