from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 이 계약을 청약할 때 반려견의 건강상태를 판단할 수 있는 기초자료(건강진 단서 사본 등)에 따라 승낙한 경우에 건강진단서 사본 '
 '등에 명기되어 있는 사항으 로 보험금 지급사유가 발생하였을 때(계약자 또는 피보험자가 회사에 제출한 기초 자료의 내용 중 중요사항을 '
 '고의로 사실과 다르게 작성한 때에는 이 특별약관을 해지할 수 있습니다) 5'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 72},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000394',
              'chunk_char_len': 192,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
