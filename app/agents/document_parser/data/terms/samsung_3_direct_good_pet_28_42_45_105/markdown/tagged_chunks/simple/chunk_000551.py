from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사가 보험금을 지정대리청구인에게 지급한 경우에는 그 이후 보험금 청구를 받더라\n'
 '도 회사는 이를 지급하지 않습니다.- \n'
 '- 제6조 (보험금의 청구)\n'
 '지정대리청구인은 회사가 정하는 방법에 따라 다음의 서류를 제출하고 보험금을 청구하\n'
 '여야 합니다.- 1. 청구서(회사양식)\n'
 '- 2. 사고증명서(장해진단서, 입원치료확인서 등)\n'
 '- 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증)\n'
 '- 4. 피보험자 및 지정대리청구인의 가족관계등록부(가족관계증명서)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000551',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
