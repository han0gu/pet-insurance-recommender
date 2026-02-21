from langchain_core.documents import Document

chunk = Document(
    page_content=('. 왕진료, 가입동물의 이송비, 동물병원에 가지 않고 약제만 배달되는 배달료 및<br>이와 동종의 비용<br>15. 안락사 비용, '
 '시체처치 및 해부검사, 장례비, 이장비 등 사후에 필요한 비용<br>16. 마이크로칩의 삽입비용, 각종 증빙서류의 작성비용(우송비 '
 '포함)<br>17. 과잉진료행위로 인한 비용<br>18'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001049',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
