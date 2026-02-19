from langchain_core.documents import Document

chunk = Document(
    page_content=('구 분 | 특정질병 | 분류코드 | 항목명\n'
 'KDA016 | 소화관 기능 저하 (소화관 정체 포함)\n'
 'KDA017 | 항문낭염 / 항문낭 파열\n'
 'KDA018 | 항문 주위 피부염 / 항문 주위 누공\n'
 'KEA001 | 식도 탈장\n'
 'KEA003 | 배꼽 탈장\n'
 'KEA004 | 사타구니 탈장 (서혜부 탈장 포함)\n'
 'KEA005 | 회음부 탈장\n'
 'KEA006 | 대퇴 탈장\n'
 'KEA007 | 직장탈장\n'
 'KEA008 | 기타 복부탈장\n'
 'KFA001 | 복막염\n'
 'KGA001 | 트리코모나스증\n'
 'KGA002 | 지아르디아 증\n'
 'KGA003 | 콕시듐증'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 173},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000608',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
