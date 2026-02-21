from langchain_core.documents import Document

chunk = Document(
    page_content=('한국표준질병․사인분류가 개정되는 경우는 개정된 기준에 따라 이 약관에서 보장| 하는 골절 해당 여부를 판단합니다. |  |\n'
 '| --- | --- |\n'
 '| 대상이 되는 항목 | 분류번호 특별 |\n'
 '| 두개골 및 안면골의 골절 파절(깨짐, 부러짐) 제외) | S02 (S02.5는 약 관 |\n'
 '| (치아의 | 제외) S07 |\n'
 '| 머리의 으깸손상 머리의 상세불명 | S09.9 |\n'
 '| 손상 목의 골절 | S12 |\n'
 '| 늑골, 흉골 및 흉추의 골절 | S22 별 |\n'
 '| 요추 및 골반의 골절 | S32 표 |'),
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
 'indexing': {'chunk_id': 'chunk_000963',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
