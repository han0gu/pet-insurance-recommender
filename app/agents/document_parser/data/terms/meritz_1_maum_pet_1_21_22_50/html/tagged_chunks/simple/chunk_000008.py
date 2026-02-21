from langchain_core.documents import Document

chunk = Document(
    page_content=(". 지급사유 관련 용어</h1><br><p id='12' data-category='list' "
 "style='font-size:14px'>가. 상해: 보험기간 중에 발생한 급격하고도 우연한 외래의 사고로 반려동물이 입은 상<br>해를 "
 '말합니다. 유독가스 또는 유독물질을 우연히 일시적으로 흡입, 흡수 또는 섭<br>취한 결과로 발생하는 중독 증상을 포함합니다. 그러나 '
 '세균성 음식물 중독과 상습<br>적으로 흡입, 흡수 또는 섭취한 결과로 생긴 중독증상은 이에 포함되지 않습니다.<br>나'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000008',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
