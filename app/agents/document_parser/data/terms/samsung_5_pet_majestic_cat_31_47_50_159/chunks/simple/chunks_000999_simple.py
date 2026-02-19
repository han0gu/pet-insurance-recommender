from langchain_core.documents import Document

chunk = Document(
    page_content=('15. 팔의 상세불명 부위의 골절 | T10\n'
 '16. 다리의 상세불명 부위의 골절 | T12\n'
 '17. 상세불명의 신체부위의 골절 | T14.2\n'
 '18. 척추 및 척수의 출산손상 | P11.5\n'
 '19. 출산손상으로 인한 두개골골절 | P13.0\n'
 '20. 두개골의 기타 출산손상 | P13.1\n'
 '21. 대퇴골의 출산손상 | P13.2\n'
 '22. 기타 긴뼈의 출산손상 | P13.3\n'
 '23. 출산손상으로 인한 쇄골의 골절 | P13.4\n'
 '24. 기타 골격 부분의 출산손상 | P13.8\n'
 '분류항목 분류번호 25. 상세불명의 골격의 출산손상 P13.9'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 151},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000999',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
