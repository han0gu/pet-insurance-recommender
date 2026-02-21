from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>\uf000 회사는 제1조(보험금의 지급사유)에서 정한 반려동물양육자금Ⅱ(질병사망)을 "
 '지<br>급한 때에는 그 지급사유가 발생한 때부터 이 특별약관은 소멸되며 이 특별약관<br>의 해약환급금을 지급하지 '
 '않습니다.<br>\uf000 회사는 제1조(보험금의 지급사유)에서 정하지 않는 사유로 피보험자가 사망하였<br>을 경우에는 이 특별약관 '
 '계약도 소멸되며 회사는 "보험료 및 해약환급금 산출방<br>법서"에서 정하는 바에 따라 피보험자의 사망 당시 이 특별약관의 '
 '계약자적립액<br>및 미경과보험료를 계약자에게'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000625',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
