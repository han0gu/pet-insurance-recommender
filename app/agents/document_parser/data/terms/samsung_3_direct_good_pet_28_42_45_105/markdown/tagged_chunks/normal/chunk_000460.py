from langchain_core.documents import Document

chunk = Document(
    page_content=('우체국, 신협, 새마을금고 등이 공제계약을 취급합니다.| 피보험자가 부담한 의료비 × | <지급보험금 계산방법> 다른 계약이 없을 때 이 '
 '계약의 지급보험금 다른 계약이 없는 것으로 하여 각각 계약의 지급보험금의 합계액 |\n'
 '| --- | --- |\n'
 '② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한'),
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
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000460',
              'chunk_char_len': 185,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
