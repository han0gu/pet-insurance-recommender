from langchain_core.documents import Document

chunk = Document(
    page_content=('정하지 않은 사항은 특별약관 일반사항을 따릅니다. 특별약관 일반사항에서도 정하지 않\n'
 '은 사항은 보통약관을 따릅니다. 다만, 보통약관 제10조(환급금의 중도인출), 제11조(만기환급금의 지급)은 제외합니다.- 110 '
 '-4-3. [갱신형] 반려묘 사망위로금 특별약관# 제1조 (보험금의 지급사유)① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하 '
 '「보험기간」 이라 합니다) 중\n'
 '에 제3항에서 정한 보장개시일(책임개시일) 이후에 보험증권에 기재된 반려묘가 보험'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000574',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
