from langchain_core.documents import Document

chunk = Document(
    page_content=('령에 정한 경우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없이 수집, 이용, 조회 또는 제공하지 않습니다. 다만, 회사는 이 '
 '특별약관의 체결, 유지, 보험금 지급 등을 위하여 위 관계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 보험회# 사 및 '
 '보험관련단체 등에 개인정보를 제공할 수 있습니다.② 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 합니다.# 제45조 (준거법)이 '
 '특별약관은 대한민국 법에 따라 규율되고 해석되며, 약관에서 정하지 않은 사항은 ｢금'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000265',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
