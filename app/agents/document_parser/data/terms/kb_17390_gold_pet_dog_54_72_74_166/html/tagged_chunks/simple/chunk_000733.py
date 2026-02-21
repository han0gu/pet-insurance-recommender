from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>제5조(특별약관의 소멸)<br>피보험자가 사망하였을 경우에는 이 특별약관 계약도 소멸되며 회사는 "
 '"보험료 및<br>해약환급금 산출방법서"에서 정하는 바에 따라 피보험자의 사망 당시 이 특별약관의</p><br><p id=\'53\' '
 "data-category='paragraph' style='font-size:16px'>계약자적립액 및 미경과보험료를 계약자에게 "
 "지급합니다.</p><br><h1 id='54' style='font-size:16px'>제6조(준용규정)</h1><br><p "
 "id='55'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000733',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
