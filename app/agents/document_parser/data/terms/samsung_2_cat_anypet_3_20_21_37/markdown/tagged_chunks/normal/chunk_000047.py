from langchain_core.documents import Document

chunk = Document(
    page_content=('을 하지 않은 때에는 계약자는 계약이 성립한 날부터 3개월 이내에 계약을 취소할 수 있습니다.【자필서명】 날인(도장을 찍음) 및 '
 '전자서명법 제2조 제2호에 따른 전자서명을 포함합니다.제3항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게 '
 '돌려드리며, 보험\n'
 '료를 받은 기간에 대하여 보험개발원이 공시하는 보험계약대출이율을 연단위 복리로 계산한 금액\n'
 '을 더하여 지급합니다.# 제18조(계약의 무효)계약을 맺을 때에 보험사고가 이미 발생하였을 경우 이 계약은 무효로 합니다. 다만, 회사의 '
 '고의 또'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000047',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
