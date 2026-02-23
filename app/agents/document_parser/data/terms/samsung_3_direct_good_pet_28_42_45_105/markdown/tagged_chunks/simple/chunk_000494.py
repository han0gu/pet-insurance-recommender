from langchain_core.documents import Document

chunk = Document(
    page_content=('만원을 초과하는 경우에 한하여 그 초과하는 배상책임 손해에 대한 금액을 보상한\n'
 '-- 88 -# 보상합니다.3. 제3조(보상하는 손해) 제2항 제2호 다.목 또는 라.목의 비용 : 이 비용과 제1호에\n'
 '의한 보상액의 합계액을 보상한도액 내에서 보상합니다.# 제8조 (손해방지의무)# ① 보험사고가 생긴 때에는 계약자 또는 피보험자는 아래의 '
 '사항을 이행하여야 합니다.- 1. 손해의 방지 또는 경감을 위하여 노력하는 일(피해자에 대한 응급처치, 긴급호송\n'
 '- 또는 그 밖의 긴급조치를 포함합니다)'),
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
 'indexing': {'chunk_id': 'chunk_000494',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
