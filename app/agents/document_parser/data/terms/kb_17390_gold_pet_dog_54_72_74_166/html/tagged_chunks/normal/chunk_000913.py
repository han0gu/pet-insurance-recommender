from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만,</p><br><p id='92' data-category='paragraph' style='font-size:14px'>이미 "
 '보험금 지급사유가 발생한 경우에는 이에 대한 보험금은 지급합니다.<br>\uf000 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 '
 '취지를 계약자에게 통지하고<br>보통약관 제1절 일반조항 제34조(해약환급금) 제1항에 따른 해약환급금을 지급합</p><br><h1 '
 "id='93' style='font-size:14px'>니다.</h1><p id='94' data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000913',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
