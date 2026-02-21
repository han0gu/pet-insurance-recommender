from langchain_core.documents import Document

chunk = Document(
    page_content=('이 끝나는 날 이전까지 중요사항 변경내역(갱신보험료 변경\n'
 '제외) 및 자동갱신 의사를 확인하는 내용 등을 서면(등기우\n'
 '편 등), 전화(음성녹음), 전자문서, 휴대전화 문자메시지\n'
 '또는 이에 준하는 전자적 의사표시 등으로 2회 이상 알려드\n'
 '리며, 자동갱신 의사가 확인되는 경우, 갱신일에 갱신일 현\n'
 '재의 약관 등으로 갱신됩니다. 다만, 계약자가 자동갱신을\n'
 '원하지 않는 경우에는 갱신일에 변경 전 계약은 만료됩니\n'
 '다.\uf000 제3항에도 불구하고 회사가 계약자의 자동갱신 의사를\n'
 '확인하지 못한 경우(계약자와 연락두절 등으로 회사 안내가'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000538',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
