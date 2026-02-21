from langchain_core.documents import Document

chunk = Document(
    page_content=('손해(수의사의 소견 및 처방에 의한 경우도 동일) 및 그로 인하여 가중된 손해\n'
 '8. 국가 및 지방자치단체의 명령 또는 법률에 의한 살처분 또는 이와 유사한 사태\n'
 '9. 원인이 어떠한 경우에도 반려견에 대한 사료제공 또는 급수 등 기본적인 관리에 대\n'
 '한 태만\n'
 '10. 동물보호법 위반 등 동물학대에 기인하는 손해\n'
 '11. 사망사실을 명확하게 입증할 수 없는 실종, 행방불명 등- \n'
 '# 제 3조 (보험금의 청구)① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.- 1. 보험금 청구서(회사 양식)'),
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
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000469',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
