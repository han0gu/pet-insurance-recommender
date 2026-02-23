from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>이 계약은 대한민국 법에 따라 규율되고 해석되며, 약관에서 정하지 않은 사항은 "
 "금융<br>소비자보호에 관한 법률, 상법, 민법 등 관계 법령을 따릅니다.</p><h1 id='180' "
 "style='font-size:16px'>제51조(예금보험에 의한 지급보장)</h1><br><p id='181' "
 "data-category='paragraph' style='font-size:14px'>회사가 파산 등으로 인하여 보험금 등을 지급하지 "
 '못할 경우에는 예금자보호법에서 정 별<br>표<br>하는 바에'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000330',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
