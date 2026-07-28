/**
 * IrisLensSDK Android - LensAdapter
 *
 * 렌즈 선택 RecyclerView 어댑터
 *
 * @version 1.0.0
 */
package com.irislenssdk.demo.lens

import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import com.irislenssdk.demo.R

/**
 * 렌즈 선택 어댑터
 */
class LensAdapter(
    private val onLensSelected: (LensData) -> Unit
) : ListAdapter<LensData, LensAdapter.LensViewHolder>(LensDiffCallback()) {

    // 현재 선택된 렌즈 ID
    private var selectedLensId: String = NoLens.ID

    // 아이템 레이아웃 — 세로(가로 스크롤 레일)는 item_lens, 가로 모드 우측 세로 리스트는 item_lens_land.
    // viewType으로 그대로 사용하므로 전환 시 기존 ViewHolder가 재사용되지 않고 새로 inflate된다
    // (RecycledViewPool을 수동으로 비울 필요 없음).
    private var itemLayoutRes: Int = R.layout.item_lens

    /**
     * 아이템 레이아웃 전환 (세로 ↔ 가로).
     *
     * @param layoutRes [R.layout.item_lens] 또는 [R.layout.item_lens_land].
     *                  두 레이아웃은 lensContainer/lensImage/lensName id를 공유해야 한다.
     */
    fun setItemLayout(layoutRes: Int) {
        if (itemLayoutRes == layoutRes) return
        itemLayoutRes = layoutRes
        notifyDataSetChanged()
    }

    override fun getItemViewType(position: Int): Int = itemLayoutRes

    /**
     * 뷰홀더
     */
    class LensViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val imageView: ImageView = itemView.findViewById(R.id.lensImage)
        val nameText: TextView = itemView.findViewById(R.id.lensName)
        val container: View = itemView.findViewById(R.id.lensContainer)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): LensViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(viewType, parent, false)
        return LensViewHolder(view)
    }

    override fun onBindViewHolder(holder: LensViewHolder, position: Int) {
        val lens = getItem(position)

        // 이름 설정
        holder.nameText.text = lens.name

        // 썸네일 설정
        if (lens.id == NoLens.ID) {
            // "없음" 아이콘
            holder.imageView.setImageResource(R.drawable.ic_lens_off)
            holder.imageView.scaleType = ImageView.ScaleType.CENTER_INSIDE
        } else {
            // 렌즈 썸네일
            lens.thumbnail?.let { bitmap ->
                holder.imageView.setImageBitmap(bitmap)
                holder.imageView.scaleType = ImageView.ScaleType.CENTER_CROP
            } ?: run {
                holder.imageView.setImageResource(R.drawable.ic_lens_placeholder)
                holder.imageView.scaleType = ImageView.ScaleType.CENTER_INSIDE
            }
        }

        // 선택 상태 표시
        val isSelected = lens.id == selectedLensId
        updateSelectionState(holder, isSelected)

        // 클릭 리스너
        holder.container.setOnClickListener {
            val previousSelected = selectedLensId
            selectedLensId = lens.id

            // 이전 선택 아이템과 새 선택 아이템 갱신
            val previousPosition = currentList.indexOfFirst { it.id == previousSelected }
            if (previousPosition >= 0) {
                notifyItemChanged(previousPosition)
            }
            notifyItemChanged(position)

            onLensSelected(lens)
        }
    }

    /**
     * 선택 상태 UI 업데이트
     */
    private fun updateSelectionState(holder: LensViewHolder, isSelected: Boolean) {
        val context = holder.itemView.context

        if (isSelected) {
            // 선택됨: 보라색 테두리
            val drawable = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setStroke(6, context.getColor(R.color.purple_500))
                setColor(Color.TRANSPARENT)
            }
            holder.imageView.background = drawable
            holder.nameText.setTextColor(context.getColor(R.color.purple_500))
        } else {
            // 선택 안됨: 회색 테두리
            val drawable = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setStroke(2, Color.parseColor("#40FFFFFF"))
                setColor(Color.TRANSPARENT)
            }
            holder.imageView.background = drawable
            holder.nameText.setTextColor(Color.WHITE)
        }
    }

    /**
     * 외부에서 선택 변경 (동기화용)
     */
    fun setSelectedLens(lensId: String) {
        val previousSelected = selectedLensId
        selectedLensId = lensId

        val previousPosition = currentList.indexOfFirst { it.id == previousSelected }
        val newPosition = currentList.indexOfFirst { it.id == lensId }

        if (previousPosition >= 0) {
            notifyItemChanged(previousPosition)
        }
        if (newPosition >= 0) {
            notifyItemChanged(newPosition)
        }
    }

    /**
     * DiffUtil 콜백
     */
    class LensDiffCallback : DiffUtil.ItemCallback<LensData>() {
        override fun areItemsTheSame(oldItem: LensData, newItem: LensData): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: LensData, newItem: LensData): Boolean {
            return oldItem == newItem
        }
    }
}
