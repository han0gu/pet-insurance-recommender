from langchain_core.documents import Document

chunk = Document(
    page_content=('- 합니다)\n'
 '3. 피보험자 본인 또는 배우자와 생계를 같이 하고, 보험증권에 기재된 주택의 주민등록상 동거중인 동거 친족(민법 제 777조)4. '
 '피보험자 본인 또는 배우자와 생계를 같이하는 별거 중인 미혼자녀<관련법규># [민법 제777조(친족의 범위)에서 규정한 친족의 범위]# '
 ': 8촌 이내의 혈족, 4촌 이내의 인척, 배우자# 제 6조 (보험금을 지급하지 않는 사유)① 회사는 아래의 사유로 보험금 지급사유가 '
 '발생한 때에는 보험금을 지급하지 않습니다.- 1. 계약자 및 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
