from langchain_core.documents import Document

chunk = Document(
    page_content=('보통약관 및 이 특별약관에서 정한 보장개시일 이전에 발생한 질병에 대하여 계약<br>을 무효로 하는 경우에도 제2조(특별면책(회사가 '
 '보험금을 지급하지 않는)조건의<br>내용) 제1항에서 정한 특정질병에 대하여 회사가 보험금을 지급하지 않는 조건으로<br>체결한 후 '
 "보장개시일 이전에 동일한 특정질병이 발생한 경우에는 계약을 무효로<br>하지 않습니다.</p><br><p id='116' "
 "data-category='list'></p><br><p id='117' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001445',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
