from langchain_core.documents import Document

chunk = Document(
    page_content=('① 「호스피스∙완화의료 및 임종과정에 있는 환자의 연명의료결정에 관한 법률」에 따른 연명의료중단 등 결정 및 그 이행으로 피보험자가 '
 "사망하는 경우 연명의료중단 등 결 정 및 그 이행은 제1조(보험금의 지급사유) '사망'의 원인 및 '사망보험금' 지급에 영향 을 미치지 "
 '않습니다. ② 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 못할 때는 보험수익자와 회사가 함께 '
 '제3자를 정하고 그 제3자의 의견에 따를 수 있 습니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 64},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000325',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
