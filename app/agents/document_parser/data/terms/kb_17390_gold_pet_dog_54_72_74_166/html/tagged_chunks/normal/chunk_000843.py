from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험설계사 등의 행위가 없었다 하더라도 계약자 또는 피보험자가<br>사실대로 알리지 않거나 부실한 사항을 알렸다고 인정되는 '
 '경우에는 계약을<br>해지할 수 있습니다.<br>제1항에 의한 계약의 해지가 손해발생 전에 이루어진 경우에는 보통약관 '
 '제1절<br>일반조항 제34조(해약환급금) 제1항에 의한 해약환급금을 계약자에게 지급합<br>니다.<br>제1항 제1호에 의한 계약의 '
 "해지가 보험금 지급사유 발생 후에 이루어진 경우에</p><br><p id='16' data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000843',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
