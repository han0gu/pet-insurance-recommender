from langchain_core.documents import Document

chunk = Document(
    page_content=('장해판정기준1) ‘외모’란 얼굴(눈, 코, 귀, 입 포함), 머리, 목을 말한다.\n'
 '2) ‘추상(추한 모습)장해’라 함은 성형수술(반흔(흉터)성형술, 레이저치\n'
 '료 등 포함)을 시행한 후에도 영구히 남게 되는 상태의 추상(추한 모습)\n'
 '을 말한다.\n'
 '3) ‘추상(추한 모습)을 남긴 때’라 함은 상처의 흔적, 화상 등으로 피부의\n'
 '변색, 모발의 결손, 조직(뼈, 피부 등)의 결손 및 함몰 등으로 성형수술\n'
 '을 하여도 더 이상 추상(추한 모습)이 없어지지 않는 경우를 말한다.'),
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
