from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 일반금융소비자에게 청약을 권유하거나 일반금융 소비자가 설명을 요청하는 경우 보험상품에 관한 중요한 사 항을 계약자가 '
 '이해할 수 있도록 설명하고 계약자가 이해하 였음을 서명(「전자서명법」 제2조 제2호에 따른 전자서명 을 포함), 기명날인 또는 녹취 등을 '
 '통해 확인받아야 하며, 설명서를 제공하여야 합니다. \uf000 설명서, 약관, 계약자 보관용 청약서 및 보험증권의 제 공 사실에 관하여 '
 '계약자와 회사간에 다툼이 있는 경우에는 회사가 이를 증명하여야 합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 80},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000167',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
