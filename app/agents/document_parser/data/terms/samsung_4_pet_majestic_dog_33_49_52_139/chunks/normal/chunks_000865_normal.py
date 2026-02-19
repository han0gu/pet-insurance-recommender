from langchain_core.documents import Document

chunk = Document(
    page_content=('17 | 신장\n'
 '18 | 부신\n'
 '19 | 요관, 방광 및 요도\n'
 '20 | 음경\n'
 '21 | 질 및 외음부\n'
 '22 | 전립선\n'
 '23 | 유방(유선 포함)\n'
 '24 | 자궁[자궁체부(자궁몸통) 포함]\n'
 '25 | 자궁체부(자궁몸통)(제왕절개술을 받은 경우에 한함)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 136},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['urinary', 'other', 'other']},
 'indexing': {'chunk_id': 'chunk_000865',
              'chunk_char_len': 134,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
