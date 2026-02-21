from langchain_core.documents import Document

chunk = Document(
    page_content=('할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합\n'
 '니다.# 제 3조 (보상하는 손해)- ① 회사는 대한민국 내에서 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간\n'
 '- 」이라 합니다) 중에 보험증권에 기재된 피보험자의 반려견의 행위에 기인하는 우연\n'
 '- 한 사고(이하「사고」라 합니다)로 인하여 타인의 신체에 피해를 입히거나 타인 소유\n'
 '- 의 반려동물에 손해를 입혀 법률상의 배상책임을 부담함으로써 입은 손해(이하「배상\n'
 '- 책임손해」라 합니다)를 이 특별약관에 따라 보상합니다.'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000482',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
