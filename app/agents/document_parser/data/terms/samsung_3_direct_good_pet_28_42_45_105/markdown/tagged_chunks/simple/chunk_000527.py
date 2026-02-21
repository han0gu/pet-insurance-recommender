from langchain_core.documents import Document

chunk = Document(
    page_content=('우체국, 신협, 새마을금고 등이 공제계약을 취급합니다.이 계약의 보상책임액\n'
 '손해액 ×\n'
 '다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 '
 '제1항에 의한\n'
 '지급보험금 결정에는 영향을 미치지 않습니다.# 제6조 (보험금을 지급하지 않는 사유)① 회사는 아래의 사유를 원인으로 하여 생긴 손해는 '
 '보상하지 않습니다.- 1. 보통약관 제5조 (보험금을 지급하지 않는 사유) 제1항\n'
 '- 2. 피보험자의 치매를 제외한 정신적 기능장해, 선천성 뇌질환 및 심신상실'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000527',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
