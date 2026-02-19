from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[약관의 중요한 내용] 금융소비자 보호에 관한 법률 제19조(설명의무) 등에서 정한 다음의 내용을 말합니다.\n'
 '- 보험금 지급제한 사유 및 지급절차\n'
 '- 청약의 철회에 관한 사항\n'
 '- 계약의 해지 및 해제'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 52},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000227',
              'chunk_char_len': 118,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
