from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 증가된 위험과 관계없이 발생한 보험금 지<br>급사유에 관해서는 원래대로 지급합니다.</p><br><p id='141' "
 "data-category='paragraph' style='font-size:14px'>예 시</p><br><p id='142' "
 "data-category='paragraph' style='font-size:14px'>비례 보상 예시</p><br><table "
 "id='143' style='font-size:14px'><thead></thead><tbody><tr><td>보험기간 "
 '중</td><td>직업의'),
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
