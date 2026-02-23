from langchain_core.documents import Document

chunk = Document(
    page_content=('및 안면골의 골절 파절(깨짐, 부러짐) 제외)</td><td>S02 (S02.5는 약 '
 '관</td></tr><tr><td>(치아의</td><td>제외) S07</td></tr><tr><td>머리의 으깸손상 머리의 '
 '상세불명</td><td>S09.9</td></tr><tr><td>손상 목의 '
 '골절</td><td>S12</td></tr><tr><td>늑골, 흉골 및 흉추의 골절</td><td>S22 '
 '별</td></tr><tr><td>요추 및 골반의 골절</td><td>S32 표</td></tr><tr><td>어깨 및 위팔의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001705',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
