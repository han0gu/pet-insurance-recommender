from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[이미 발생한 보험금 지급사유에 대한 보험금 지급]\n'
 '계약자 또는 피보험자가 보험금 청구에 관한 서류를 변조하여 보험금을 청구한 경우, 회사는 그 사 실을 안 날부터 1개월 이내에 계약을 '
 '해지할 수 있습니다. 다만, 이 경우에도 회사는 실제 발생한 보험금 지급사유에 대해서는 보험금을 지급합니다.\n'
 '② 회사가 제1항에 따라 이 특별약관을 해지한 경우 회사는 그 취지를 계약자에게 통 지하고 이 특별약관의 해약환급금을 지급합니다.\n'
 '제26조 (회사의 손해배상책임)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 75},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000439',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
