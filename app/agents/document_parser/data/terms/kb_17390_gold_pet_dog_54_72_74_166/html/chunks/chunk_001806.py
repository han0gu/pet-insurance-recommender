from langchain_core.documents import Document

chunk = Document(
    page_content=('질병명</td></tr><tr><td></td></tr><tr><td>고관절 탈구 / 아탈구</td></tr><tr><td>고관절이형성증 '
 '골절 (골반)</td></tr><tr><td>골절 (기타)</td></tr><tr><td>골절 '
 '(전지)</td></tr><tr><td>골절 '
 '(후지)</td></tr><tr><td>관절염</td></tr><tr><td>근염</td></tr><tr><td>대퇴골두허혈성괴사</td></tr><tr><td>생후 '
 '골유합 부전</td></tr><tr><td>슬관절 탈구 /'),
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
