from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[신의성실의 원칙]\n'
 '계약관계의 당사자는 권리를 행사하거나 의무를 이행할 때 상대방의 정당한 이익을 배려해야 하고 신뢰를 저버리지 않도록 행동해야 한다는 '
 '원칙을 말합니다. ※ 민법 제2조(신의성실) ①권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다.\n'
 '② 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게 유리하게 해석합니다. ③ 회사는 보험금을 지급하지 않는 사유 등 계약자나 '
 '피보험자에게 불리하거나 부담을 주는 내용은 확대하여 해석하지 않습니다.\n'
 '제43조 (설명서 교부 및 보험안내자료 등의 효력)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 48},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000165',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
