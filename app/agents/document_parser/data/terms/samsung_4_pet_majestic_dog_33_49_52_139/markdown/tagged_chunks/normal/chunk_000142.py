from langchain_core.documents import Document

chunk = Document(
    page_content=('금 등을 지급받지 못할 수 있습니다.# 제 42조 (약관의 해석)① 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하여야 하며 '
 '계약자에 따라 다\n'
 '르게 해석하지 않습니다.# <용어풀이># [신의성실의 원칙]계약관계의 당사자는 권리를 행사하거나 의무를 이행할 때 상대방의 정당한 이익을 '
 '배려해야 하고\n'
 '신뢰를 저버리지 않도록 행동해야 한다는 원칙을 말합니다.\n'
 '※ 민법 제2조(신의성실) ①권리의 행사와 의무의 이행은 신의에 좇아 성실히 하여야 한다.- ② 회사는 약관의 뜻이 명백하지 않은 '
 '경우에는 계약자에게 유리하게 해석합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000142',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
