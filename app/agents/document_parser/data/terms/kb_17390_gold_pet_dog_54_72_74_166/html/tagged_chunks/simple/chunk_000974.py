from langchain_core.documents import Document

chunk = Document(
    page_content=('. 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비 등 사후에 필요한 비용<br>16. 마이크로칩의 삽입비용, 각종 증빙서류의 '
 '작성비용(우송비 포함)<br>17. 과잉진료행위로 인한 비용<br>18. 아포퀠(Apoquel) 등의 JAK inhibitor(Janus '
 "kinase inhibitor) 약물과 사이토<br>포인트(Cytopoint)</p><br><p id='163' "
 "data-category='paragraph' style='font-size:14px'>19"),
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
 'indexing': {'chunk_id': 'chunk_000974',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
