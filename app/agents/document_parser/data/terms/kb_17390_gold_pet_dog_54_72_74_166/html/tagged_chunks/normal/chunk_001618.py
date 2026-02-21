from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그 중 심장에서 가까운 쪽<br>부터 중수지관절, 제1지관절(근위지관절) 및 제2지관절(원위지관절)이<br>라 부른다.<br>5) '
 '‘손가락을 잃었을 때’라 함은 첫째 손가락에서는 지관절부터 심장에서<br>가까운 쪽에서, 다른 네 손가락에서는 '
 '제1지관절(근위지관절)부터(제1지<br>관절 포함) 심장에서 가까운 쪽으로 손가락이 절단되었을 때를 말한다.<br>6) ‘손가락뼈 일부를 '
 '잃었을 때’라 함은 첫째 손가락의 지관절, 다른 네 손<br>가락의 제1지관절(근위지관절)부터 심장에서 먼 쪽으로 손가락 뼈의 '
 '일부<br>가 절단된 경우를'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001618',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
