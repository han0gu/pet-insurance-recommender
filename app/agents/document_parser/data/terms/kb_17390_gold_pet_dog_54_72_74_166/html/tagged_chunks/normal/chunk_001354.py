from langchain_core.documents import Document

chunk = Document(
    page_content=('. 반<br>\uf000 회사는 제1항 및 제2항의 규정에 의한 지급기일 내에 이 특별약관의 보험금을 지 려<br>급하지 않았을 때에는 '
 '그 지급기일의 다음날부터 지급일까지의 기간에 대하여 "보 동<br>험금을 지급할 때의 적립이율 계산(【별표2】참조)"에서 정한 이율로 '
 "계산한 금액 물<br>을 더하여 지급합니다.</p><br><p id='231' data-category='paragraph' "
 "style='font-size:16px'>제8조(특별약관의 보험료)</p><br><p id='232' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_001354',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
