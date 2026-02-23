from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 때 회 통약<br>사는 해지 전 발생한 보험금 지급사유를 이유로 부활(효력회복)을 거절하지 '
 '않습<br>관<br>니다.<br>\uf000 제1항에서 정한 해지계약의 부활(효력회복)이 이루어진 경우라도 계약자 또는 피<br>보험자가 '
 '최초계약 청약시(2회 이상 부활이 이루어진 경우 종전 모든 부활 청약<br>포함) 제14조(계약 전 알릴 의무)를 위반한 경우에는 '
 '제16조(알릴 의무 위반의 효<br>과)가 적용됩니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
