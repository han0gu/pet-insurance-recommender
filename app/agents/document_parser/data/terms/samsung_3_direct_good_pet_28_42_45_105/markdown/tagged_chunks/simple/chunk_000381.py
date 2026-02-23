from langchain_core.documents import Document

chunk = Document(
    page_content=('사례에 비추어 보험수익자에게 매우 불합리하게 합의를 하는 것을 의미합니다.# 제 27조 (특별약관의 재가입에 관한 사항)① 계약이 다음 '
 '각 호의 조건을 충족하고 계약자가 제4항에 따라 재가입의사를 표시한\n'
 '때에는 특별약관 일반사항의 제19조(특별약관의 성립) 및 제21조(약관 교부 및 설명\n'
 '의무 등)를 준용하여 회사가 정한 절차에 따라 계약자는 기존 계약에 이어 재가입할\n'
 '수 있으며, 이 경우 회사는 기존계약의 가입 이후 발생한 반려동물의 상해 또는 질병'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000381',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
