from langchain_core.documents import Document

chunk = Document(
    page_content=("또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것</h1><p id='118' data-category='paragraph' "
 "style='font-size:14px'>제4조(특별약관의 소멸)<br>피보험자가 사망하였을 경우에는 이 특별약관 계약도 소멸되며 회사는 "
 '"보험료 및<br>해약환급금 산출방법서"에서 정하는 바에 따라 피보험자의 사망 당시 이 특별약관의</p><br><h1 id=\'119\' '
 "style='font-size:14px'>계약자적립액 및 미경과보험료를 계약자에게 지급합니다.</h1><p id='120'"),
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
 'indexing': {'chunk_id': 'chunk_000435',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
