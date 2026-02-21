from langchain_core.documents import Document

chunk = Document(
    page_content=('였음을 서명(「전자서명법」 제2조 제2호에 따른 전자서명\n'
 '을 포함), 기명날인 또는 녹취 등을 통해 확인받아야 하며,\n'
 '설명서를 제공하여야 합니다.\n'
 '\uf000 설명서, 약관, 계약자 보관용 청약서 및 보험증권의 제\n'
 '공 사실에 관하여 계약자와 회사간에 다툼이 있는 경우에는\n'
 '회사가 이를 증명하여야 합니다.\n'
 '\uf000 보험설계사 등이 모집과정에서 사용한 회사 제작의 보험\n'
 '안내자료의 내용이 약관의 내용과 다른 경우에는 계약자에'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000133',
              'chunk_char_len': 227,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
