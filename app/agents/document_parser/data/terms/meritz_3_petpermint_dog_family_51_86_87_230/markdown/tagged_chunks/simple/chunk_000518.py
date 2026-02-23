from langchain_core.documents import Document

chunk = Document(
    page_content=('재의 약관 등으로 갱신됩니다. 다만, 계약자가 자동갱신을\n'
 '원하지 않는 경우에는 갱신일에 변경 전 계약은 만료됩니\n'
 '다.\uf000 제5항에도 불구하고 회사가 계약자의 자동갱신 의사를\n'
 '확인하지 못한 경우(계약자와 연락두절 등으로 회사 안내가\n'
 '계약자에게 도달하지 못한 경우 포함)에는 갱신일 현재의\n'
 '약관 등으로 갱신됩니다. 다만, 계약자는 갱신일 현재의 약\n'
 '관 등에 대해 90일 이내에 그 계약을 취소할 수 있습니다.# 제15조(특별약관의 무효)이 특별약관을 가입할 때에 보험사고가 이미 '
 '발생하였을 경'),
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
 'indexing': {'chunk_id': 'chunk_000518',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
