from langchain_core.documents import Document

chunk = Document(
    page_content=('손해액 × 다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액\n'
 '② 이 계약이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 의무보험에서 보 상되는 금액(피보험자가 가입을 하지 않은 경우에는 '
 '보상될 것으로 추정되는 금액)을 차감한 금액을 손해액으로 간주하여 제1항에 의한 보상할 금액을 결정합니다. ③ 피보험자가 다른 계약에 '
 '대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '제11조 (대위권)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 89},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000580',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
