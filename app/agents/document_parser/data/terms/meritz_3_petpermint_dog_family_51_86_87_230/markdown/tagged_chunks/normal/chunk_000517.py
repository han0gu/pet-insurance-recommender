from langchain_core.documents import Document

chunk = Document(
    page_content=('안내합니다.\n'
 '\uf000 제3항 및 제4항에도 불구하고 법령 및 표준약관 변경으\n'
 '로 보장내용 등이 변경되어 약관이 개정되는 경우 보험기간183이 끝나는 날 이전까지 중요사항 변경내역(갱신보험료 변경\n'
 '제외) 및 자동갱신 의사를 확인하는 내용 등을 서면(등기우\n'
 '편 등), 전화(음성녹음), 전자문서, 휴대전화 문자메시지\n'
 '또는 이에 준하는 전자적 의사표시 등으로 2회 이상 알려드\n'
 '리며, 자동갱신 의사가 확인되는 경우, 갱신일에 갱신일 현\n'
 '재의 약관 등으로 갱신됩니다. 다만, 계약자가 자동갱신을'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000517',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
