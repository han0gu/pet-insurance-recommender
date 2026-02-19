from langchain_core.documents import Document

chunk = Document(
    page_content=('【위법계약】\n'
 '금융소비자보호에 관한 법률 제47조에서 정한 적합성원 칙, 적정성원칙, 설명의무, 불공정영업행위 금지 또는 부당권유행위 금지를 위반한 '
 '계약을 말합니다.\n'
 '【제척기간】\n'
 '권리관계를 확정하기 위하여 어떤 종류의 권리에 대하여 법률이 정하고 있는 존속 기간을 말하며, 이 기간이 지 나면 해당 권리는 '
 '소멸됩니다.\n'
 '제33조(중대사유로 인한 해지)\n'
 '\uf000 회사는 아래와 같은 사실이 있을 경우에는 안 날부터 1 개월 이내에 계약을 해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 80},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000148',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
