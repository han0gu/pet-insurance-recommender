from langchain_core.documents import Document

chunk = Document(
    page_content=('![image](/image/placeholder)\n'
 '제1번(1th)\n'
 '제2번 (2th)\n'
 '제3번 (3th)\n'
 '흉골병\n'
 '(Manubrium)\n'
 '제4번 (4th)\n'
 '흉골 흉골체\n'
 '(Sternum) (Body) 제5번 (5th)\n'
 '검상돌기 늑골\n'
 '(Xiphoid process) 제6번 (6th) (Ribs)\n'
 '늑연골 제7번 (7th)\n'
 '(Costal cartilage)\n'
 '제8번 (8th)\n'
 '제9번 (9th)\n'
 '척추연골늑\n'
 '(Vertebrochondral ribs) 제10번 (10th)\n'
 '늑골절흔\n'
 '(Costal notch) 제11번(11th)\n'
 '제12번(12th)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000784',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
