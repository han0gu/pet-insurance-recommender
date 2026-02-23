from langchain_core.documents import Document

chunk = Document(
    page_content=('에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 전액 부담합니다. 성\n'
 '\uf000 피보험자가 보험기간 중 사망하고, 그 후에 제3조(천식지속상태의 정의 및 진단 특\n'
 '확정)에서 정한 "천식지속상태"를 직접적인 원인으로 사망한 사실이 확인된 경우 약- 91 -KB 금쪽같은 '
 '펫보험(강아지)(무배당)(26.01) 91에는 그 사망일을 진단 확정일로 보고 제1조(보험금의 지급사유)에 해당하는 경'),
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
