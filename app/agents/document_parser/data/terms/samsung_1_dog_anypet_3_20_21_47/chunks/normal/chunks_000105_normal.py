from langchain_core.documents import Document

chunk = Document(
    page_content=('제34조(약관의 해석)\n'
 '① 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하여야 하며 계약자에 따라 다르게 해석하지 않습니다.\n'
 '【신의성실의 원칙】 계약관계의 당사자는 권리를 행사하거나 의무를 이행할 때 상대방의 정당한 이익을 배려하고 신뢰에 어긋나지 않도록 '
 '행동해야 한다는 원칙을 말합니다.\n'
 '【관련법규】\n'
 '< 「민법」 제2조(신의성실))> ① 권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 19},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000105',
              'chunk_char_len': 227,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
